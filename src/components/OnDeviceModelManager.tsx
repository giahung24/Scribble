import { useState, useEffect, useCallback } from 'react';
import { getOndeviceModels, downloadOndeviceModel, type OndeviceModelStatus } from '../lib/api';

interface Props {
    lang: 'vi' | 'en';
    modelSize: 'small' | 'medium' | 'large-v3';
    translateEnabled?: boolean;
}

type Row = {
    key: string;
    label: string;
    present: boolean;
    kind: 'whisper' | 'nllb';
    size?: string;
};

export default function OnDeviceModelManager({ lang, modelSize, translateEnabled = false }: Props) {
    const [status, setStatus] = useState<OndeviceModelStatus | null>(null);
    const [busy, setBusy] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        try {
            const r = await getOndeviceModels();
            setStatus(r);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const handleDownload = useCallback(
        async (kind: 'whisper' | 'nllb', size: string | undefined, rowKey: string) => {
            setBusy(rowKey);
            setError(null);
            try {
                const r = await downloadOndeviceModel(kind, size);
                if (!r.ok) {
                    setError(r.error || (lang === 'vi' ? 'Tải thất bại' : 'Download failed'));
                } else {
                    await refresh();
                }
            } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
            } finally {
                setBusy(null);
            }
        },
        [lang, refresh],
    );

    if (!status) {
        return (
            <div className="setting-group setting-group--full">
                <div className="setting-hint">{lang === 'vi' ? 'Đang kiểm tra model...' : 'Checking models...'}</div>
            </div>
        );
    }

    const rows: Row[] = [];
    const seen = new Set<string>();
    const addWhisper = (size: 'small' | 'medium' | 'large-v3') => {
        if (seen.has(size)) return;
        seen.add(size);
        rows.push({
            key: size,
            label: `Whisper ${size}`,
            present: status.whisper[size],
            kind: 'whisper',
            size,
        });
    };
    addWhisper(modelSize);
    addWhisper('large-v3'); // always — used by upload/batch
    if (translateEnabled) {
        rows.push({ key: 'nllb', label: 'NLLB (translate)', present: status.nllb, kind: 'nllb' });
    }

    return (
        <div className="setting-group setting-group--full">
            <div className="setting-hint">
                {lang === 'vi'
                    ? 'Model on-device được lưu cục bộ. Tải file có thể lớn và chạy nền (chỉ cập nhật khi tải xong).'
                    : 'On-device models are stored locally. Downloads can be large and run in the background (status updates only when finished).'}
            </div>
            {rows.map((row) => (
                <div
                    key={row.key}
                    style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '4px 0' }}
                >
                    <span>
                        {row.label}{' '}
                        <span style={{ color: row.present ? 'var(--success, #2e7d32)' : 'var(--warning, #b26a00)' }}>
                            {row.present
                                ? lang === 'vi' ? '✓ đã tải' : '✓ ready'
                                : lang === 'vi' ? '• chưa tải' : '• missing'}
                        </span>
                    </span>
                    {!row.present && (
                        <button
                            type="button"
                            className="btn btn-secondary"
                            disabled={busy !== null}
                            onClick={() => handleDownload(row.kind, row.size, row.key)}
                        >
                            {busy === row.key
                                ? lang === 'vi' ? 'Đang tải...' : 'Downloading...'
                                : lang === 'vi' ? 'Tải' : 'Download'}
                        </button>
                    )}
                </div>
            ))}
            {error && <div className="setting-warning">{error}</div>}
        </div>
    );
}
